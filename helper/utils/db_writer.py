import sqlalchemy as db
engine = db.create_engine('postgresql+psycopg2://sao:ghx!28s4§kikv$9\"asd5@campaneo.l3s.uni-hannover.de:5432/sao')

def create_connection():
    connection = engine.connect()
    metadata = db.MetaData()
    metadata.reflect(bind=connection)
    return connection, metadata

def insert_result(model, dataset, demand_type, map_h, map_w, c, p, t, n_residual, n_epoch, lr, mse, mae, rmse, comment,
                  train_flag=True, meta_flag=True):
    conn, meta = create_connection()
    result_table = meta.tables['pretraining_result']
    # insert into database
    if train_flag:
        query = result_table.insert().values(model=model, dataset=dataset, demand_type=demand_type, map_h=map_h, map_w=map_w, closeness=c, period=p, trend=t,
                                                    n_residual_unit=n_residual, n_epoch=n_epoch, lr=lr,
                                                    train_mse=mse, train_mae=mae, train_rmse=rmse, comment=comment)
    else:
        if meta_flag:
            query = result_table.insert().values(model=model, dataset=dataset, demand_type=demand_type, map_h=map_h,
                                                 map_w=map_w, closeness=c, period=p, trend=t,
                                                 n_residual_unit=n_residual, n_epoch=n_epoch, lr=lr,
                                                 test_mse=mse, test_mae=mae, test_rmse=rmse, comment=comment)
        else:
            query = result_table.update().where((result_table.c.model==model) & (result_table.c.dataset==dataset) &
                                            (result_table.c.demand_type==demand_type) & (result_table.c.map_h==map_h) &
                                            (result_table.c.map_w==map_w) & (result_table.c.closeness==c)
                                            & (result_table.c.period==p) & (result_table.c.trend==t) &
                                            (result_table.c.n_residual_unit==n_residual) & (result_table.c.comment==comment) &
                                            (result_table.c.lr==lr)).values(test_mse=mse, test_mae=mae, test_rmse=rmse)
    result = conn.execute(query)
    conn.close()

def insert_result_new(model, dataset, map_h, map_w, c, p, t, n_residual, n_epoch, lr, mse, mae, rmse, comment,
                  train_flag=True, meta_flag=True):
    conn, meta = create_connection()
    result_table = meta.tables['mmd_new_new']
    # insert into database
    if train_flag:
        query = result_table.insert().values(model=model, dataset=dataset, map_h=map_h, map_w=map_w, closeness=c, period=p, trend=t,
                                                    n_residual_unit=n_residual, n_epoch=n_epoch, lr=lr,
                                                    train_mse=mse, train_mae=mae, train_rmse=rmse, comment=comment)
    else:
        if meta_flag:
            query = result_table.insert().values(model=model, dataset=dataset, map_h=map_h,
                                                 map_w=map_w, closeness=c, period=p, trend=t,
                                                 n_residual_unit=n_residual, n_epoch=n_epoch, lr=lr,
                                                 test_mse=mse, test_mae=mae, test_rmse=rmse, comment=comment)
        else:
            query = result_table.update().where((result_table.c.model==model) & (result_table.c.dataset==dataset) &
                                                (result_table.c.map_h==map_h) & (result_table.c.map_w==map_w) &
                                                (result_table.c.closeness==c) & (result_table.c.period==p) &
                                                (result_table.c.trend==t) & (result_table.c.n_residual_unit==n_residual) &
                                                (result_table.c.comment==comment) &
                                                (result_table.c.lr==lr)).values(test_mse=mse, test_mae=mae, test_rmse=rmse)
    result = conn.execute(query)
    conn.close()


def insert_result_pretrain_v2(model, dataset, map_h, map_w, c, p, t, n_residual, n_epoch, lr, mse, mae, rmse, comment,
                               table_name, train_flag=True, meta_flag=True):

    conn, meta = create_connection()
    result_table = meta.tables[table_name]
    # insert into database
    if train_flag:
        query = result_table.insert().values(model=model, dataset=dataset, map_h=map_h, map_w=map_w, closeness=c,
                                             period=p, trend=t,
                                             n_residual_unit=n_residual, n_epoch=n_epoch, lr=lr,
                                             train_mse=mse, train_mae=mae, train_rmse=rmse, comment=comment)
    else:
        if meta_flag:
            query = result_table.insert().values(model=model, dataset=dataset, map_h=map_h,
                                                 map_w=map_w, closeness=c, period=p, trend=t,
                                                 n_residual_unit=n_residual, n_epoch=n_epoch, lr=lr,
                                                 test_mse=mse, test_mae=mae, test_rmse=rmse, comment=comment)
        else:
            query = result_table.update().where(
                (result_table.c.model == model) & (result_table.c.dataset == dataset) &
                (result_table.c.map_h == map_h) & (result_table.c.map_w == map_w) &
                (result_table.c.closeness == c) & (result_table.c.period == p) &
                (result_table.c.trend == t) & (result_table.c.n_residual_unit == n_residual) &
                (result_table.c.comment == comment) &
                (result_table.c.lr == lr)).values(test_mse=mse, test_mae=mae, test_rmse=rmse)
    result = conn.execute(query)
    conn.close()